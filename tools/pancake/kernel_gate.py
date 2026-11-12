#!/usr/bin/env python3
"""kernel_gate.py - MOD_PLAN "Project features discovered" item 9: a go / no-go for an end-to-end screen of a kernel item,
from kernel-level numbers (minutes, low noise) instead of engine starts (hours for +-1%: F21, F9, A33).

QUESTION   A kernel item saves X ms per decode window (or Y% of a prefill).  What does that buy end to end on the short and
           the edit window, is it above the bar (2%, MOD_PLAN 16 row 8), and how many engine starts would an A/B need to
           resolve it?  The answer is a verdict per item:
             GO        a measured kernel saving gives >= the bar on a window: run the end-to-end confirm.
             NO-GO     even the optimistic bound is below the bar (or the measured saving is): do not spend a screen; an
                       exact item may still ship on its parity test and kernel number if it costs nothing.
             MEASURE   only an estimate or a bound reaches the bar: measure the kernel (the "missing" column) first.
INPUT      the kernel's saving, from the microbench of the item, plus where it sits in the window:
             --expdec FILE [--expdec-new FILE]   expdec_bench output (cell lines `dev T shape gu/d choice all_us gu_us down_us
                         GB/s pct identical`; --base-choice / --new-choice / --t pick the cells; a DIFFERENT cell is refused)
             --q2rows FILE        q2_rows_bench output (legacy vs fast ns/row; the N-thread block is the pool)
             --stage FILE --prefill-tok-s N   hip_gdn_stage_bench output (STAGE / A31 lines) and the prompt speed
             --saved-ms X / --saved-frac F    a saving worked out by hand (e.g. from two rocprofv3 tables)
             --profile FILE       the kernel table of the decode window: gpu_profile.py's summary.txt (top 20 kernels) or a
                         rocprofv3 `--stats` csv; default bench/results/2026-11-09-gpuprof-defer/summary.txt (120 windows, deferral on).
                         It gives each kernel's ms per window and calls per window (the number of windows is read from the
                         once-per-layer kernels).
MODEL      decode items go through the A6 engine-chain model, tools/pancake/time_ceiling.py window_ms2, at the fast-hip preset
           (deferral + FUSE_PHASES): a kernel is on one lane, and the model says how much of its saving lands in the window.
           lane `router` / `mixer` = GPU work before the doorbell (all of it, or most of it, lands); `hit` = the VRAM expert
           kernels, hidden behind the CPU rows (little lands on edits, most on short windows where the CPU finishes first);
           `cpu` = the CPU expert pool.  A GPU speedup while the CPU lane is the critical path buys ~0, as the model says.
           Prefill items are Amdahl on the GPU timeline (share x saved fraction).  The two decode windows are the plan's
           (short 24.8 ms: GPU 15.9, CPU 5.2, draft 2.2; edit 37.4 ms: GPU 15.8, CPU 17.4, draft 2.5), the edit one calibrated
           as in A6 with one fitted number, the dispersion of the per-layer misses (no routing trace is needed): deferral gives
           the A6 33.76 ms; serial (39.2 against A6's 39.26) and the preset's FUSE_PHASES (33.4 against 33.37) are not fitted.  The SHORT WINDOW IS NOT VALIDATED (A6: no short routing
           trace): read its percentages as an order of magnitude.  Percentages are on the measured preset window (short 24.8 ms,
           edit 34.42 ms = 33.37 + the A6 bench-vs-profile constant 1.05), not on the profile window.
STARTS     the A/B is paired (one engine start per arm per repetition, 2 starts per pair); the paired SD of one pair is
           taken as 3.9% of base (the median of the 62 attempts of 2026-11-11 as given for this tool; `table` prints the
           figure recomputed from bench/results/2026-11-11-gpu/*.jsonl, which has only one spec with >= 4 pairs).  Pairs needed: the smallest n with t(n-1) * SD / sqrt(n) <= gain (95% interval excludes 0 at the
           expected gain; 80% power in brackets).  One start is ~2.1 min (median 118-126 s of the 62 attempts).
RUN        python tools/pancake/kernel_gate.py table [--profile FILE] [--bar 2] [--paired-sd 3.9]
           python tools/pancake/kernel_gate.py item F7 --q2rows /tmp/q2.txt
           python tools/pancake/kernel_gate.py item F12 --expdec /tmp/expdec.txt        (choices dec:s0 -> dec:s1)
           python tools/pancake/kernel_gate.py item FLAT --expdec base.txt --expdec-new flat.txt --base-choice dec --new-choice dec
           python tools/pancake/kernel_gate.py item A31 --stage /tmp/a31-stage.txt --prefill-tok-s 2580
Items: F7 F12 F21 F9 A33 A37 FLAT A36 A31 BFLA.  Python only; it runs no benchmark and reads no model.
"""
import argparse
import csv
import io
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "probes"))
import gdn_chunk64_rule as g64  # noqa: E402  (STAGE / A31 line parser)
import time_ceiling as tc  # noqa: E402  (the A6 engine-chain model: window_ms2)
import trace_common  # noqa: E402  (kernel name shortener)

BAR_PCT = 2.0                 # MOD_PLAN 16 row 8: end to end only when kernel gain x exposed share >= 2% of the window
PAIRED_SD_PCT = 3.9           # median paired SD (% of base) of one A/B pair, 2026-11-11 data
MIN_PER_START = 2.1           # minutes per engine start + measurement (median 118-126 s)
E2E_MAX_STARTS = 60           # ~2 hours of starts: above it an end-to-end screen cannot decide the item
NL = tc.rp.NL                 # 48 MoE layers
DEFAULT_PROFILE = ROOT / "bench/results/2026-11-09-gpuprof-defer/summary.txt"
DEFAULT_JSONL = ROOT / "bench/results/2026-11-11-gpu"

# ---- the A6 chain model, at the fast-hip preset ---------------------------------------------------------------------
# p of time_ceiling.window_ms2: GPU work before the doorbells `pre` (mixer share `mu`), VRAM experts `E` (hidden behind the CPU),
# pool barriers `ps`, combine sync, c = ms of CPU per distinct missed expert, promotion DMA contention (all A6 --validate defaults).
A6_P = dict(pre=15.75, mu=0.65, E=4.3, ps=0.040, sync=0.040, c=0.0446, alpha=0.5, dma_per=1.4 / 12.0)
HEAVY = 0.4                   # share of a layer's CPU time in rows kept on time (BELOW 0.1)
SDEF = 0.091                  # ms per deferred layer (A6 fit to +15.8% deferral)
FUSE_CUT = A6_P["ps"] / 2     # STRATA_POOL_FUSE_PHASES (in the preset): one pool phase instead of two
A6_DEFER_EDIT_MS = 33.76      # A6: edit window under deferral (the calibration point of the dispersion)
# the measured window the percentages are on, and the CPU/draft/promotion of the window (MOD_PLAN 16 budget table)
WINDOWS = {
    "short": dict(cpu_ms=5.2, draft=2.2, promo=26.8 * (5.2 - NL * 0.040) / (17.41 - NL * 0.040), bench_ms=24.8, validated=False),
    "edit": dict(cpu_ms=17.41, draft=2.5, promo=26.8, bench_ms=33.37 + 1.05, validated=True),
}
LANES = ("router", "mixer", "hit", "cpu")


class LaneModel:
    """window_ms2 over 200 synthetic windows of per-layer misses (lognormal around the window's mean, one fixed random
    stream).  The dispersion is fitted once so that deferral on the edit window gives the A6 figure."""

    def __init__(self, n_windows=200, seed=0, bench_ms=None):
        self.bench = {w: spec["bench_ms"] for w, spec in WINDOWS.items()}
        self.bench.update(bench_ms or {})                      # e.g. {"short": 32.6}: the denominator of the percentages
        self.z = np.random.default_rng(seed).standard_normal((n_windows, NL))
        self.sigma = self._fit_sigma()
        self.rows = {w: self._misses(w, self.sigma) for w in WINDOWS}
        self.base = {w: self._ms(w) for w in WINDOWS}

    def _misses(self, window, sigma):
        spec = WINDOWS[window]
        mean = (spec["cpu_ms"] - NL * A6_P["ps"]) / A6_P["c"] / NL      # distinct missed experts per layer
        return (mean * np.exp(sigma * self.z - sigma * sigma / 2)).tolist()

    def _fit_sigma(self):
        spec, lo, hi = WINDOWS["edit"], 0.0, 2.0
        for _ in range(40):
            mid = (lo + hi) / 2
            rows = self._misses("edit", mid)
            p = dict(A6_P, draft=spec["draft"])
            w = float(np.mean([tc.window_ms2(r, spec["promo"], p, "defer", HEAVY, SDEF) for r in rows]))
            lo, hi = (mid, hi) if w < A6_DEFER_EDIT_MS else (lo, mid)
        return (lo + hi) / 2

    def _ms(self, window, p=None, mode="defer", keep=1.0, ps_cut=FUSE_CUT):
        spec = WINDOWS[window]
        q = dict(p or A6_P, draft=spec["draft"])
        return float(np.mean([tc.window_ms2(r, spec["promo"], q, mode, HEAVY, SDEF, ps_cut, keep) for r in self.rows[window]]))

    def window_ms(self, window, mode="defer", p=None, ps_cut=FUSE_CUT):
        """Model window (ms): the preset (mode defer, FUSE_PHASES) unless told otherwise."""
        return self._ms(window, p=p, mode=mode, ps_cut=ps_cut)

    def delta_ms(self, window, lane, amount):
        """ms of window saved when the kernel on `lane` gets `amount` faster: ms of that lane's work per window, or for `cpu`
        the fraction of the CPU experts' time."""
        p, keep = dict(A6_P), 1.0
        if lane == "cpu":
            keep = 1.0 - amount
        elif lane == "hit":
            p["E"] = max(0.0, p["E"] - amount)
        elif lane in ("router", "mixer"):
            mixer, router = p["pre"] * p["mu"], p["pre"] * (1 - p["mu"])
            mixer, router = (max(0.0, mixer - amount), router) if lane == "mixer" else (mixer, max(0.0, router - amount))
            p["pre"], p["mu"] = mixer + router, mixer / (mixer + router)
        else:
            raise ValueError(f"unknown lane {lane!r} (one of {', '.join(LANES)})")
        return self.base[window] - self._ms(window, p=p, keep=keep)

    def gain_pct(self, window, lane, amount):
        """End-to-end gain (%) on the measured preset window (tok/s ratio) for that saving."""
        bench = self.bench[window]
        return (bench / (bench - self.delta_ms(window, lane, amount)) - 1.0) * 100.0

    def exposure(self, window, lane, amount):
        """Share of a GPU lane's saving that lands in the window (1.0 = all of it)."""
        return self.delta_ms(window, lane, amount) / amount if amount > 0 and lane != "cpu" else None


def prefill_gain_pct(saved_frac):
    """Amdahl on the GPU timeline: gain (%) of the prompt speed when `saved_frac` of the prefill time disappears."""
    return (1.0 / (1.0 - saved_frac) - 1.0) * 100.0


# ---- engine starts --------------------------------------------------------------------------------------------------
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120,
        2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t_crit(df):
    """Two-sided 95% Student t."""
    return _T95[df - 1] if df <= len(_T95) else 1.96 + 2.4 / df


def pairs_needed(gain_pct, sd_pct, power=False, cap=5000):
    """Smallest number of A/B pairs n whose 95% interval half-width t * SD / sqrt(n) is <= the gain (with power: the 80%
    power form, (t + 0.84) * SD / sqrt(n) <= gain).  None when more than `cap`."""
    if gain_pct <= 0:
        return None
    z = 0.8416 if power else 0.0
    for n in range(2, cap + 1):
        if (t_crit(n - 1) + z) * sd_pct / math.sqrt(n) <= gain_pct:
            return n
    return None


def starts_for(gain_pct, sd_pct):
    """(starts for the 95% interval to exclude 0 at that gain, starts for 80% power); 2 engine starts per pair."""
    out = []
    for power in (False, True):
        n = pairs_needed(gain_pct, sd_pct, power)
        out.append(None if n is None else 2 * n)
    return tuple(out)


# ---- parsers ----------------------------------------------------------------------------------------------------------
EXPDEC_ROW = re.compile(r"^(\d+) (\d+) (\w+) (\d+)/(\d+) (\S+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)% (identical|DIFFERENT)\s*$",
                        re.M)


def parse_expdec_bench(text):
    """One dict per cell line of src/kernels/expdec_bench.cpp ('#' context lines and the decomposition are skipped)."""
    rows = [dict(dev=int(m[1]), T=int(m[2]), shape=m[3], pair=f"{m[4]}/{m[5]}", choice=m[6], all_us=float(m[7]),
                 gu_us=float(m[8]), down_us=float(m[9]), gbps=float(m[10]), pct=float(m[11]), identical=m[12] == "identical")
            for m in EXPDEC_ROW.finditer(text)]
    if not rows:
        raise ValueError("no expdec_bench cell line (dev T shape gu/d choice all_us gu_us down_us GB/s pct identical)")
    return rows


def expdec_cell(rows, pair, choice, T=4, shape="real"):
    for r in rows:
        if r["pair"] == pair and r["choice"] == choice and r["T"] == T and r["shape"] == shape:
            return r
    raise ValueError(f"expdec_bench: no cell pair {pair} choice {choice} T={T} shape {shape}")


Q2_HEAD = re.compile(r"^(down|gate/up)\s+\(.*?\), (L2-hot|DRAM), [^:]*: (\d+) rows, fast vs legacy bitwise equal: (yes|NO)")
Q2_ARM = re.compile(r"^\s+(legacy|fast|dispatch|read only)\s+([\d.]+) ns/row\s+([\d.]+) Mrows/s\s+([\d.]+) GB/s of weights\s+"
                    r"([\d.]+) x floor")
Q2_THREADS = re.compile(r"^\s+-- (\d+) threads, (\d+) rows per call:")
Q2_TITLE = re.compile(r"^Q2_0 row kernels, one thread, nt=(\d+)")
Q2_EXPERT_ROWS = {"gate/up": 1280, "down": 2560}     # rows of one Q2_0 expert (MOD_PLAN 16.10: 1,382,400 B)


def parse_q2_rows_bench(text):
    """One dict per arm line of tools/pancake/probes/q2_rows_bench.cpp: nt (tokens per expert), shape (down | gate/up), ws
    (L2-hot | DRAM), threads (1, or N for the pool block), arm, ns_row, x_floor, equal."""
    rows, cur, threads, nt = [], None, 1, 1
    for line in text.splitlines():
        m = Q2_TITLE.match(line)
        if m:
            nt = int(m[1])
            continue
        m = Q2_HEAD.match(line)
        if m:
            cur, threads = dict(shape=m[1], ws=m[2], equal=m[4] == "yes"), 1
            continue
        m = Q2_THREADS.match(line)
        if m:
            threads = int(m[1])
            continue
        m = Q2_ARM.match(line)
        if m and cur:
            rows.append(dict(cur, nt=nt, threads=threads, arm=m[1], ns_row=float(m[2]), x_floor=float(m[5])))
    if not rows:
        raise ValueError("no q2_rows_bench arm line (legacy / fast ... ns/row ... x floor)")
    return rows


def q2_expert_saving(rows, threads=None):
    """Per-expert time of the legacy and the fast Q2_0 kernel from the DRAM rows (1,280 gate/up + 2,560 down rows per expert),
    on `threads` (default: the largest count present, the pool block when the bench was run with --threads)."""
    dram = [r for r in rows if r["ws"] == "DRAM"]
    if not dram:
        raise ValueError("q2_rows_bench: no DRAM rows")
    threads = threads or max(r["threads"] for r in dram)
    if not all(r["equal"] for r in rows):
        raise ValueError("q2_rows_bench: fast and legacy are not bitwise equal; F7 is not exact, do not gate it")

    def expert_us(arm):
        ns = {r["shape"]: r["ns_row"] for r in dram if r["arm"] == arm and r["threads"] == threads}
        if set(ns) != set(Q2_EXPERT_ROWS):
            raise ValueError(f"q2_rows_bench: no {arm} DRAM rows at {threads} thread(s) for both shapes")
        return sum(Q2_EXPERT_ROWS[s] * ns[s] for s in ns) / 1e3
    legacy, fast = expert_us("legacy"), expert_us("fast")
    return dict(threads=threads, nt=dram[0]["nt"], legacy_us=legacy, fast_us=fast, frac=1.0 - fast / legacy)


SUMMARY_ROW = re.compile(r"^\s*([\d.]+)%\s+([\d.]+) ms\s+(\d+)x\s+([\d.]+) us\s+(\S.*?)\s*$")
# once-per-layer / once-per-window kernels of a decode window, to read the number of windows from a trace
WINDOW_MARKERS = (("wait_flag_copy_i32_kernel", NL), ("combine_cpu", NL), ("gdn_step_commit_layers_kernel", 1))


class KernelTable:
    """Kernel name -> calls, total ms (and min us when the table has it) of a decode phase, and the windows it covers."""

    def __init__(self, kernels, windows=None):
        self.kernels = kernels
        self.windows = windows or self._infer_windows()
        self.listed_only = False    # gpu_profile.py's summary lists the top 20 kernels only

    def _infer_windows(self):
        for marker, per in WINDOW_MARKERS:
            calls = sum(v["calls"] for k, v in self.kernels.items() if marker in k)
            if calls:
                return max(1, round(calls / per))
        raise ValueError("cannot read the number of decode windows from this table (no wait_flag_copy / combine_cpu kernel): "
                         "pass windows=")

    def _sel(self, rx):
        pat = re.compile(rx)
        return [v for k, v in self.kernels.items() if pat.search(k)]

    def ms(self, rx):
        """ms per window of the kernels whose name matches rx."""
        sel = self._sel(rx)
        if not sel:
            raise KeyError(f"kernel /{rx}/ is not in the table (gpu_profile.py's summary lists only the top 20: use a rocprofv3 "
                           f"--stats csv for the rest)")
        return sum(v["ms"] for v in sel) / self.windows

    def calls(self, rx):
        """Calls per window of the kernels matching rx."""
        return sum(v["calls"] for v in self._sel(rx)) / self.windows

    def unlisted_ms(self):
        """Upper bound (ms per window) of a kernel the table does not list: the smallest listed one (0 for a full table)."""
        return min(v["ms"] for v in self.kernels.values()) / self.windows if self.listed_only else 0.0


def parse_kernel_table(text, windows=None):
    """gpu_profile.py's summary.txt (`  8.8%   346.9 ms   4080x   85.0 us  expdec_gu_kernel<22, 1, 4>`) or a rocprofv3 --stats
    csv ("Name","Calls","TotalDurationNs",...,"MinNs",...).  Names are shortened (namespace and argument list dropped)."""
    kernels, listed_only = {}, False
    if text.lstrip().startswith('"Name"'):
        for r in csv.DictReader(io.StringIO(text)):
            k = trace_common.short(r["Name"])
            v = kernels.setdefault(k, dict(calls=0, ms=0.0, min_us=None))
            v["calls"] += int(r["Calls"])
            v["ms"] += float(r["TotalDurationNs"]) / 1e6
            if r.get("MinNs"):
                v["min_us"] = float(r["MinNs"]) / 1e3
    else:
        for line in text.splitlines():
            m = SUMMARY_ROW.match(line)
            if m:
                kernels[m[5]] = dict(calls=int(m[3]), ms=float(m[2]), min_us=None)
        listed_only = True
    if not kernels:
        raise ValueError("no kernel rows (gpu_profile.py summary.txt or a rocprofv3 --stats csv)")
    t = KernelTable(kernels, windows)
    t.listed_only = listed_only
    return t


# ---- the end-to-end evidence that exists (abtest jsonl) ----------------------------------------------------------------
def kept_runs(recs):
    """The attempt abtest keeps per (arm, rep): the first clean one, else the quietest."""
    by = {}
    for r in recs:
        by.setdefault((r["arm"], r["rep"]), []).append(r)
    return {k: next((x for x in rs if x["clean"]), min(rs, key=lambda x: x["noise"])) for k, rs in by.items()}


def _vals(kept, arm, case):
    return [row["decode_tps"] for rep in sorted(r for a, r in kept if a == arm) for row in kept[(arm, rep)]["rows"]
            if row["case"] == case]


def paired_gain(recs, arm, case, base="base"):
    """abtest's summary line for arm vs base on one case: (mean %, 95% half-width %, n pairs, SD of a pair in % of base)."""
    kept = kept_runs(recs)
    a, b = _vals(kept, arm, case), _vals(kept, base, case)
    n = min(len(a), len(b))
    if n < 2:
        return None
    d = [x - y for x, y in zip(a[:n], b[:n])]
    bm = sum(b) / len(b)
    sd = statistics.stdev(d)
    return (statistics.mean(d) / bm * 100, t_crit(n - 1) * sd / math.sqrt(n) / bm * 100, n, sd / bm * 100)


def paired_sd_from_jsonl(recs_by_file, min_pairs=4):
    """Median SD (% of base) of the paired differences of every (file, arm, case) with >= min_pairs pairs, and the list."""
    sds = []
    for name, recs in recs_by_file.items():
        arms = list(dict.fromkeys(r["arm"] for r in recs))
        for arm in arms[1:]:
            for case in ("short", "edit"):
                g = paired_gain(recs, arm, case, arms[0])
                if g and g[2] >= min_pairs:
                    sds.append((name, arm, case, g[2], g[3]))
    return (statistics.median(s[4] for s in sds) if sds else None), sds


def load_jsonl_dir(path):
    out = {}
    for f in sorted(Path(path).glob("*.jsonl")):
        out[f.name] = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return out


# ---- the items ----------------------------------------------------------------------------------------------------------
@dataclass
class Candidate:
    """One kernel item.  point / bound are the saving: ms per window for the GPU lanes, the fraction of the CPU experts' time
    for `cpu`, the fraction of the prefill time for kind prefill.  point None = no number yet; measured = the point comes from
    a kernel measurement (not a static estimate); bound = the most the notes consider possible."""
    key: str
    name: str
    switch: str
    kind: str                 # decode | prefill
    lane: str                 # router | mixer | hit | cpu (decode), gpu (prefill)
    point: object
    bound: float
    measured: bool
    basis: str
    missing: str
    kernel_ms: object = None  # ms per window of the kernels the item changes (to express the break-even as a kernel speedup)
    e2e: object = None        # text: the end-to-end evidence that exists


@dataclass
class Evidence:
    """What the caller measured for one item (all optional): `item` is its key, the other items ignore it."""
    item: object = None
    expdec: object = None
    expdec_new: object = None
    base_choice: object = None
    new_choice: object = None
    T: int = 4
    q2rows: object = None
    stage: object = None
    prefill_tok_s: object = None


GU22, GU16 = r"expdec_gu_kernel<22,", r"expdec_gu_kernel<16,"
DOWN = r"native_down_multi_kernel<42|expdec_down_kernel<42"
PAIR_GU = {"16/42": GU16, "22/42": GU22}         # expdec_bench pair -> the profile kernel that carries its gate/up calls


def expdec_saved_ms(ev, prof, fields, choices, pairs=None):
    """ms per window saved by the new choice over the old, from expdec_bench cells (per layer call, us) x calls per window of
    that pair's kernel (gate/up and all_us fields) or of the down kernel (down_us: mean over the pairs); refuses a cell that
    is not bit-identical.  `pairs` = {expdec pair: profile kernel regex}, default both Q2_0-down pairs."""
    base_c, new_c = ev.base_choice or choices[0], ev.new_choice or choices[1]
    new_rows = ev.expdec_new or ev.expdec
    total, down = 0.0, []
    for pair, rx in (pairs or PAIR_GU).items():
        b, n = expdec_cell(ev.expdec, pair, base_c, ev.T), expdec_cell(new_rows, pair, new_c, ev.T)
        if not n["identical"]:
            raise ValueError(f"expdec_bench: {pair} {new_c} is DIFFERENT from the reference bit for bit; not exact, not gated")
        for f in fields:
            if f == "down_us":
                down.append(b[f] - n[f])
            else:
                total += prof.calls(rx) * (b[f] - n[f]) / 1e3
    if down:
        total += prof.calls(DOWN) * statistics.mean(down) / 1e3
    return total


def build_candidates(prof, ev=None, e2e=None):
    """The items decided so far, with the numbers of the batch2 / batch3 notes and the profile; `ev` replaces the missing ones."""
    ev = ev or Evidence()
    e2e = e2e or {}

    def use(key):
        return ev if ev.item == key else Evidence()
    gu22, gu16, down = prof.ms(GU22), prof.ms(GU16), prof.ms(DOWN)
    hit_kernels = gu22 + gu16 + down
    out = []

    e = use("F7")
    # F7  kern-cpu.md: single thread fast/legacy x1.25 (gate/up) x1.47 (down); the pool is DDR4-bound at one token per expert
    c = Candidate("F7", "Q2_0 AVX2 fast kernel", "STRATA_Q2_FAST=1", "decode", "cpu", None, 0.22, False,
                  "q2_rows_bench 1 thread: gate/up x1.25, down x1.47 (kern-cpu.md); pool at ~27 of 33 GB/s DDR4: any CPU kernel "
                  "saves <= ~22% of CPU-expert time", "q2_rows_bench --threads 8 --nt 1 and --nt 2 on an idle box (A48): the "
                  "pool's ns/row, legacy vs fast")
    if e.q2rows:
        s = q2_expert_saving(e.q2rows)
        if s["threads"] > 1:
            c.point, c.measured, c.missing = s["frac"], True, ""
        else:
            c.bound = min(c.bound, s["frac"])
        c.basis = (f"q2_rows_bench nt={s['nt']} at {s['threads']} thread(s): per expert {s['legacy_us']:.0f} -> {s['fast_us']:.0f} us "
                   f"({s['frac'] * 100:.0f}% less)" + ("" if s["threads"] > 1 else "; a single thread is not the pool: bound only"))
    out.append(c)

    e = use("F12")
    # F12 kern-cpu.md: only the identity is checked; boxwrench's -29/-35% is another engine (GLM, 7900 XT)
    c = Candidate("F12", "IQ2 sign expansion", "STRATA_IQ_SIGN_FAST=1", "decode", "hit", None, 0.35 * (gu22 + gu16), False,
                  "no gfx1100 timing (iq_sign_fast_test: identity only); bound = boxwrench's -35% of the IQ2_S + IQ2_XXS gate/up "
                  f"kernels ({gu22 + gu16:.2f} ms per window, deferral profile)",
                  "expdec_bench ... 16/42,22/42 def,dec:s0,dec:s1 (gu_us at T=4, both `identical`)", gu22 + gu16)
    if e.expdec:
        c.point, c.measured, c.missing = expdec_saved_ms(e, prof, ("gu_us",), ("dec:s0", "dec:s1")), True, ""
        c.basis = "expdec_bench gate/up phase, dec:s0 -> dec:s1, x calls per window"
    out.append(c)

    # F21 kern-hip.md: ~4 us a plan copy x 48; the copy part of wait_flag_copy_i32 is 5.44 us at best
    calls = prof.calls("wait_flag_copy_i32_kernel")
    c = Candidate("F21", "plan copy 256 threads x 16 B", "STRATA_WAIT_COPY256=1", "decode", "hit", calls * 4e-3, calls * 5.44e-3,
                  False, "static: ~4 us saved per plan copy (11 dependent PCIe reads -> 2), 48 copies; bound = the whole 5.44 us "
                  "minimum of wait_flag_copy_i32 (kern-hip.md)",
                  "gpu_profile.py --env STRATA_WAIT_COPY256=1: wait_flag_copy_i32_kernel MinNs (5.44 us before)")
    out.append(c)

    # F9  kern-hip.md: route_multi 9.7 + doorbell_publish 6.9 + 6.8 us gap -> one launch of about route_multi's length
    c = Candidate("F9", "router publish fast path", "STRATA_ROUTE_FAST=1", "decode", "router", NL * 12e-3, NL * (9.7 + 6.9 + 6.8 - 9.7) * 1e-3,
                  False, "static: ~12 us a layer (3 launches -> 1); bound = 23.4 - 9.7 us (the fused launch cannot be shorter than "
                  "route_multi); the doorbell rings ~10 us earlier too (kern-hip.md)",
                  "gpu_profile.py --env STRATA_ROUTE_FAST=1: route_publish_fast us per call vs route_multi + doorbell_publish + gap")
    out.append(c)

    e = use("A33")
    # A33 swiglu: gap expdec_gu -> swiglu 9.2 us x 34 IQ2_S layers, plus the SwiGLU kernel (not in the top 20: < the 20th row)
    gap = 34 * 9.2e-3
    c = Candidate("A33", "fused gate/up + SwiGLU", "STRATA_EXPDEC_SWIGLU=1", "decode", "hit", gap, gap + prof.unlisted_ms(), False,
                  "static: gap 9.2 us x 34 IQ2_S layers (~1% of a window, kern-hip.md); bound adds the whole SwiGLU kernel (below "
                  "the 20th row of the profile)", "gpu_profile.py or expdec_bench all_us with STRATA_EXPDEC_SWIGLU=1 vs 0 "
                  "(per layer us saved)", None, e2e.get("A33"))
    if e.expdec:
        c.point, c.measured, c.missing = expdec_saved_ms(e, prof, ("all_us",), ("dec", "dec")), True, ""
        c.basis = "expdec_bench all_us (gate/up + SwiGLU + q8_1 + down), swiglu off -> on, x calls per window"
    out.append(c)

    e = use("A37")
    # A37 kern-hip.md: x16 saves 25-45% of the IQ2_XXS gate/up kernel before the lost occupancy; the kernel is 1.7% of GPU time
    c = Candidate("A37", "IQ2_XXS codebook copies in LDS", "STRATA_IQ_LUT_DUP=16", "decode", "hit", None, 0.45 * gu16, False,
                  f"static: x16 saves 25-45% of expdec_gu<16> ({gu16:.2f} ms per window) before the halved occupancy (kern-hip.md)",
                  "expdec_bench 16/42 def,dec:l0,dec:l8,dec:l16 (gu_us; revert if < 10%) and the LDS bank counters", gu16)
    if e.expdec:
        c.point, c.measured, c.missing = expdec_saved_ms(e, prof, ("gu_us",), ("dec:l0", "dec:l16"), {"16/42": GU16}), True, ""
        c.basis = "expdec_bench 16/42 gate/up phase, dec:l0 -> dec:l16, x calls per window"
    out.append(c)

    e = use("FLAT")
    # flat loads, A36: the VRAM-hit kernels (13% of GPU kernel time before deferral); EXPDEC_V2 reports -20..-25% per kernel on gfx1201
    flat_missing = ("expdec_bench 16/42,22/42 gu_us and down_us of a -DSTRATA_HIP_GLOBAL_WEIGHTS=1 build vs the default, "
                    "`identical` in both")
    c = Candidate("FLAT", "global instead of flat weight loads", "STRATA_HIP_GLOBAL_WEIGHTS=1 (build)", "decode", "hit", None,
                  0.25 * hit_kernels, False,
                  f"compile only (flat_load 13,119 -> 108); bound = -25% of the hit kernels ({hit_kernels:.2f} ms per window; "
                  "EXPDEC_V2's gfx1201 figure, none on gfx1100)", flat_missing, hit_kernels)
    if e.expdec:
        c.point, c.measured, c.missing = expdec_saved_ms(e, prof, ("gu_us", "down_us"), ("dec", "dec")), True, ""
        c.basis = "expdec_bench gate/up and down phases, default build -> flat-load build, x calls per window"
    out.append(c)
    out.append(Candidate("A36", "weight-major hit experts", "(not built)", "decode", "hit", None, 0.25 * hit_kernels, False,
                         f"bound = a 25% faster hit kernel ({hit_kernels:.2f} ms per window); hit_critical.py on the 11-09 trace gave "
                         "0.30 / 1.58 / 2.47% of the profiled wall (pre-deferral); MonoMoE's tile mapping is not a layout",
                         "hit_critical.py on gpu_profile.py traces with deferral on, short and edit", hit_kernels))

    e = use("A31")
    # A31 kern-gdn.md: no 7900 XT number; the share bound is the 3060's CUDA 6.0% (only recurrence share measured), halving it
    share_bound, frac_bound = 0.06, 0.5
    basis = "no gfx1100 number; share bound = the RTX 3060's CUDA 6.0% of a 41.8K prompt, saving bound = halving (the rule of A37 note)"
    missing = "hip_gdn_stage_bench 4096 16384 32768 (rec_out_ms vs the WMMA floor) and the 32K prompt speed"
    if e.stage and 32768 in e.stage and e.prefill_tok_s:
        s = e.stage[32768]
        share = g64.share_from_stage(s["rec_out_ms"], 32768, e.prefill_tok_s, 36) / 100
        if s["floor_ms"]:      # chunk 64 adds ~12% WMMAs: it cannot go below 1.12 x the floor
            frac_bound = max(0.0, 1.0 - 1.12 * s["floor_ms"] / s["rec_out_ms"])
        share_bound, missing = share, "a built chunk-64 kernel (the stage bench only bounds it)"
        basis = (f"stage bench T=32768: rec_out {s['rec_out_ms']:.2f} ms/layer = {share * 100:.2f}% of a prompt at "
                 f"{e.prefill_tok_s:g} tok/s; saving bound {frac_bound * 100:.0f}% (1 - 1.12 x floor / rec_out)")
    out.append(Candidate("A31", "GDN prefill chunk 64", "(not built)", "prefill", "gpu", None, share_bound * frac_bound, False,
                         basis, missing))

    # BFLA dense kern-gdn.md / PANCAKE_STATUS P2b: 128K prompt 386.9 off -> 771.2 dense tok/s (measured, one run each)
    c = Candidate("BFLA", "BFLA dense (exact fused attention)", "STRATA_BFLA_DENSE=1", "prefill", "gpu", 1 - 386.9 / 771.2,
                  1 - 386.9 / 771.2, True,
                  "e2e measured, qwen35 engine, 128K prompt: 386.9 -> 771.2 tok/s (1 run each); the kernel model (283 s -> 96 s "
                  "of a 338 s prompt) would say +124%: 19 s of other time", "64K off baseline (dense 1,058 tok/s has none)")
    out.append(c)

    return out


@dataclass
class Result:
    cand: Candidate
    verdict: str
    reason: str
    gains: dict          # window -> (point %, bound %)
    exposure: dict       # window -> share of the saving that lands (GPU lanes)
    starts: object       # (95% interval, 80% power) at the expected gain, or None
    breakeven: dict      # window -> saving (unit of the item) that reaches the bar, or None


def _windows_of(c):
    return ("prefill",) if c.kind == "prefill" else ("short", "edit")


def _gain(model, c, window, amount):
    if c.kind == "prefill":
        return prefill_gain_pct(amount)
    return model.gain_pct(window, c.lane, amount)


def _breakeven(model, c, window, bar):
    """Smallest saving (item's unit) that gives `bar`% on the window, by bisection; None when even the largest is short."""
    hi = 0.999 if (c.kind == "prefill" or c.lane == "cpu") else 0.5 * model.bench[window]
    if _gain(model, c, window, hi) < bar:
        return None
    lo = 0.0
    for _ in range(18):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _gain(model, c, window, mid) < bar else (lo, mid)
    return hi


def evaluate(c, model, bar=BAR_PCT, sd=PAIRED_SD_PCT):
    """Verdict of one item (see the module doc): the gain is taken on the better of its windows."""
    bound = max(c.bound, c.point or 0.0)
    gains, expo, be = {}, {}, {}
    for w in _windows_of(c):
        gains[w] = (None if c.point is None else _gain(model, c, w, c.point), _gain(model, c, w, bound))
        expo[w] = model.exposure(w, c.lane, c.point if c.point is not None else bound) if c.kind == "decode" else None
        be[w] = _breakeven(model, c, w, bar)

    def best(i):
        w = max((w for w in gains if gains[w][i] is not None), key=lambda w: gains[w][i])
        return gains[w][i], w
    if c.measured and c.point is not None:
        ref, w = best(0)
        verdict = "GO" if ref >= bar else "NO-GO"
        reason = f"the measured saving gives {ref:+.1f}% on {w} ({'>=' if ref >= bar else '<'} the {bar:g}% bar)"
    elif best(1)[0] < bar:
        ref, w = best(0) if c.point is not None else best(1)
        verdict = "NO-GO"
        reason = f"even the bound gives only {best(1)[0]:+.1f}% (< the {bar:g}% bar)"
    else:
        ref, w = best(0) if c.point is not None else best(1)
        verdict = "MEASURE"
        reason = f"the estimate / bound reaches {best(1)[0]:+.1f}% on {best(1)[1]} (>= the {bar:g}% bar) but nothing is measured"
        if best(1)[1] == "short" and "edit" in gains and gains["edit"][1] < bar:
            reason += "; short window only (its model is not validated)"
    return Result(c, verdict, reason, gains, expo, starts_for(ref, sd) if ref > 0 else None, be)


# ---- output -------------------------------------------------------------------------------------------------------------
def _fmt_amount(c, x):
    if x is None:
        return "-"
    return f"{x * 100:.0f}%" if (c.lane == "cpu" or c.kind == "prefill") else f"{x:.2f}ms"


def _fmt_gain(g):
    point, bound = g
    return f"<={bound:+.1f}" if point is None else (f"{point:+.1f}" if abs(point - bound) < 0.05 else f"{point:+.1f} ({bound:+.1f})")


def _fmt_starts(s):
    if s is None:
        return "-"

    def f(x):
        return ">10000" if x is None else str(x)
    return f"{f(s[0])} ({f(s[1])})"


def render_table(results, bar, sd, bench=None):
    bench = bench or {w: spec["bench_ms"] for w, spec in WINDOWS.items()}
    head = (f"{'item':5s} {'lane':6s} {'saving pt / bound':>17s} {'short %':>13s} {'edit %':>13s} {'prefill %':>10s} "
            f"{'starts 95% (80%)':>17s}  {'verdict':8s} missing")
    lines = [head, "-" * 140]
    for r in results:
        c, g = r.cand, r.gains
        miss = c.missing if len(c.missing) < 56 else c.missing[:53] + "..."
        lines.append(f"{c.key:5s} {c.lane:6s} {_fmt_amount(c, c.point) + ' / ' + _fmt_amount(c, c.bound):>17s} "
                     f"{_fmt_gain(g['short']) if 'short' in g else '-':>13s} {_fmt_gain(g['edit']) if 'edit' in g else '-':>13s} "
                     f"{_fmt_gain(g['prefill']) if 'prefill' in g else '-':>10s} {_fmt_starts(r.starts):>17s}  {r.verdict:8s} {miss or '-'}")
    lines += ["", f"gain = end-to-end % on the measured preset window (short {bench['short']:g} ms, edit {bench['edit']:g} ms): `x (y)` = the estimate (the "
              f"bound); `<=y` = no number yet, the bound only.  Bar {bar:g}%.  starts = engine starts for the 95% interval to exclude 0 "
              f"at the expected gain (paired SD {sd:g}%; 80% power in brackets), 2 per pair, ~{MIN_PER_START:g} min each; above "
              f"{E2E_MAX_STARTS} an end-to-end screen cannot decide the item.  The short-window model is not validated (A6)."]
    return "\n".join(lines)


def render_item(r, model, bar, sd):
    c = r.cand
    out = [f"{c.key} {c.name} [{c.switch}]  lane {c.lane}, {c.kind}", f"  basis:    {c.basis}"]
    if c.e2e:
        out.append(f"  e2e so far: {c.e2e}")
    state = "measured" if c.measured else "estimate" if c.point is not None else "missing"
    out.append(f"  saving:   point {_fmt_amount(c, c.point)} ({state}), bound {_fmt_amount(c, c.bound)}"
               + (f"  (the item's kernels: {c.kernel_ms:.2f} ms per window)" if c.kernel_ms else ""))
    for w, g in r.gains.items():
        ex, be = r.exposure.get(w), r.breakeven.get(w)
        kern = ""
        if be is not None and c.kernel_ms and c.kind == "decode" and c.lane != "cpu":
            kern = f" = {be / c.kernel_ms * 100:.0f}% of the item's kernels"
        out.append(f"  {w:7s} gain {_fmt_gain(g)}%" + (f", {ex * 100:.0f}% of the saving lands" if ex is not None else "")
                   + f"; break-even saving {_fmt_amount(c, be) if be is not None else 'unreachable'}{kern}"
                   + ("   (model not validated)" if w in WINDOWS and not WINDOWS[w]["validated"] else ""))
    s = r.starts
    if s:
        out.append(f"  starts:   {_fmt_starts(s)} for the expected gain at a paired SD of {sd:g}%"
                   + (f" (~{s[0] * MIN_PER_START:.0f} min)" if s[0] and c.kind == "decode" else "")
                   + ("  -> an end-to-end screen cannot decide this" if s[0] is None or s[0] > E2E_MAX_STARTS else "")
                   + ("  [prefill SD not measured: no long* attempt in the 62]" if c.kind == "prefill" else ""))
    out.append(f"  verdict:  {r.verdict}: {r.reason}" + (f"; measure: {c.missing}" if c.missing else ""))
    return "\n".join(out)


def e2e_evidence(jsonl_dir=DEFAULT_JSONL):
    """Text of the end-to-end A/B that exists for an item (abtest jsonl), by item key: only the 6-pair confirm of spec 25
    (expdec-swiglu, A33); the other kernel arms have one start (a screen) and no interval."""
    out = {}
    path = Path(jsonl_dir) / "25-kern-hip-confirm.jsonl"
    if path.exists():
        recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        parts = []
        for case in ("short", "edit"):
            g = paired_gain(recs, "expdec-swiglu", case)
            if g:
                word = "better" if g[0] - g[1] > 0 else "worse" if g[0] + g[1] < 0 else "no difference"
                parts.append(f"{case} {g[0]:+.1f} +-{g[1]:.1f}% (n={g[2]} pairs, SD {g[3]:.1f}%) {word}")
        if parts:
            out["A33"] = "spec 25 confirm, expdec-swiglu vs base: " + "; ".join(parts)
    return out


def load_profile(path=None):
    path = Path(path) if path else DEFAULT_PROFILE
    return parse_kernel_table(path.read_text())


TAKES = {"expdec": {"F12", "A33", "A37", "FLAT"}, "q2rows": {"F7"}, "stage": {"A31"}}   # which item reads which measurement


def _read(path):
    return Path(path).read_text()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--profile", help="kernel table of the decode window (default: the 11-09 deferral profile)")
    common.add_argument("--bar", type=float, default=BAR_PCT, help="end-to-end gain (%%) an item must reach")
    common.add_argument("--paired-sd", type=float, default=PAIRED_SD_PCT, help="SD (%% of base) of one A/B pair")
    common.add_argument("--short-ms", type=float, help="short window the percentages are on (default 24.8 ms, the plan's; the "
                                                       "deferral profile had 4.3 tokens per window = ~32.6 ms at 132 tok/s)")
    sub.add_parser("table", parents=[common], help="every item decided so far, from the notes")
    it = sub.add_parser("item", parents=[common], help="one item, with your measurement")
    it.add_argument("key", choices=["F7", "F12", "F21", "F9", "A33", "A37", "FLAT", "A36", "A31", "BFLA"])
    it.add_argument("--expdec")
    it.add_argument("--expdec-new")
    it.add_argument("--base-choice")
    it.add_argument("--new-choice")
    it.add_argument("--t", type=int, default=4, help="tokens per window of the expdec_bench cells (1, 2, 4)")
    it.add_argument("--q2rows")
    it.add_argument("--stage")
    it.add_argument("--prefill-tok-s", type=float)
    it.add_argument("--saved-ms", type=float, help="measured saving, ms per window")
    it.add_argument("--saved-frac", type=float, help="measured saving, fraction of the CPU experts' time / of the prefill")
    a = ap.parse_args(argv)
    if a.cmd == "item":
        for flag, value in (("expdec", a.expdec), ("q2rows", a.q2rows), ("stage", a.stage)):
            if value and a.key not in TAKES[flag]:
                ap.error(f"{a.key} does not use --{flag} (it is read by {', '.join(sorted(TAKES[flag]))})")

    prof = load_profile(a.profile)
    ev = Evidence()
    if a.cmd == "item":
        ev = Evidence(item=a.key, expdec=parse_expdec_bench(_read(a.expdec)) if a.expdec else None,
                      expdec_new=parse_expdec_bench(_read(a.expdec_new)) if a.expdec_new else None,
                      base_choice=a.base_choice, new_choice=a.new_choice, T=a.t,
                      q2rows=parse_q2_rows_bench(_read(a.q2rows)) if a.q2rows else None,
                      stage=g64.parse_stage(_read(a.stage)) if a.stage else None, prefill_tok_s=a.prefill_tok_s)
    cands = build_candidates(prof, ev, e2e_evidence())
    if a.cmd == "item":
        cands = [c for c in cands if c.key == a.key]
        for c in cands:      # a saving worked out by hand is a measurement
            if a.saved_ms is not None and c.kind == "decode" and c.lane != "cpu":
                c.point, c.measured, c.missing = a.saved_ms, True, ""
            if a.saved_frac is not None and (c.kind == "prefill" or c.lane == "cpu"):
                c.point, c.measured, c.missing = a.saved_frac, True, ""
    model = LaneModel(bench_ms={"short": a.short_ms} if a.short_ms else None)
    results = [evaluate(c, model, a.bar, a.paired_sd) for c in cands]
    if a.cmd == "table":
        print(render_table(results, a.bar, a.paired_sd, model.bench))
        recs = load_jsonl_dir(DEFAULT_JSONL) if DEFAULT_JSONL.exists() else {}
        med, groups = paired_sd_from_jsonl(recs)
        if med is not None:
            print(f"paired SD recomputed from bench/results/2026-11-11-gpu/*.jsonl (abtest's kept attempts, groups with >= 4 pairs): "
                  f"median {med:.1f}% over {len(groups)} groups (range {min(g[4] for g in groups):.1f}-{max(g[4] for g in groups):.1f}%); "
                  f"used: {a.paired_sd:g}%")
        for r in results:
            print()
            print(render_item(r, model, a.bar, a.paired_sd))
    else:
        print(render_item(results[0], model, a.bar, a.paired_sd))
    return 0


if __name__ == "__main__":
    sys.exit(main())
