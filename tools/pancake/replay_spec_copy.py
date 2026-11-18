"""replay_spec_copy.py - offline twin of STRATA_ADAPT_SPEC_COPY (LEADS.md B147; include/strata/program/tier_spec_copy.hpp).

The adaptive tier decides each swap at a round: the INCOMING expert (a non-resident one with a high decayed count) is known only
then.  This replays a --dump-routing trace through the tier (replay_cache_policy.py's adapt_round, the daily preset: a round every
window, 32 swaps) and, at the END of every round, predicts the NEXT round's incoming experts on the CPU, "stages" them (a bounded
per-layer staging area), and counts how many of the next round's real incoming experts were already staged.  Placement never
changes (the staging only changes where the copy's bytes come from), so the tier is simulated once and the predictors are
scored against it.

Predictors (each = the tier's own selection rule run on a predicted count table; the swaps it would pick are the predicted incoming):
  bl        backlog: the counts after this round's decay (next round's counts if the next window routed nothing)
  blt       bl + the transition matrix of tier_stat_policy.hpp (this round's used experts -> next window's used experts, probability p)
  ps        persistence: bl + this window's counts once more (the next window routes like this one)
  pst       ps + the transition probability p
  la:R      bl + the next window's real routing counts, each routed entry kept with probability R (a router lookahead predictor
            with recall R; the ledger's measured recall is 58-69 % at k = 14: T-187..T-189, T-401)
  or        bl + the next window's real routing (perfect lookahead; must give 100 % - a self-check)

Parameters of the switch (bounded): --slots S staging slots per layer, --max M staged copies per round (all layers), --conf C a
predicted swap needs a gain (candidate count - victim count) >= C, --age A rounds a staged expert is kept.

Metrics per window (paired bootstrap over windows, circular blocks): hit rate = staged real incoming / real incoming; waste share =
staged copies never used / staged copies; bytes vs baseline = (staged + normal copies) bytes / normal copies bytes; saving per window
= hits x (H2D time - D2D time) per expert (--bw-h2d / --bw-d2d GB/s, --bytes per expert, ledger T-193 link 13.4 GB/s).

    python tools/pancake/replay_spec_copy.py bench/prefetch/routing.bin data/expert-profile-pancake.bin 10909
    python tools/pancake/replay_spec_copy.py real1.trace PROFILE 10909 --json out.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_cache_policy as R  # noqa: E402
import replay_fill_transition as F  # noqa: E402

# IQ2_XS preset: the bytes of one expert blob per layer format (DERIVED from the dims, not measured: hidden 2560, expert width 640,
# gate/up IQ2_S 820 B per row (34 layers), IQ2_XXS 660 (11), IQ1_M 560 (3), down Q2_0 180 B per row; T-250: Q2_0 expert 1,382,400 B).
EXPERT_BYTES = {"iq2_s": 2 * 640 * 820 + 2560 * 180, "iq2_xxs": 2 * 640 * 660 + 2560 * 180, "iq1_m": 2 * 640 * 560 + 2560 * 180}
LAYER_FORMATS = ["iq2_s"] * 34 + ["iq2_xxs"] * 11 + ["iq1_m"] * 3        # layer order unknown: the mean is what is used


def mean_bytes():
    return sum(EXPERT_BYTES[f] for f in LAYER_FORMATS) / len(LAYER_FORMATS)


def trans_p(stat):
    """Probability that each expert is used in the next round, per layer, from the Stat's transition tables (as apply_trans)."""
    nl, ne = stat.nl, stat.ne
    p = np.zeros((nl, ne), np.float32)
    s = min(max(stat.s, 0.0), 1.0)
    for l in range(nl):
        S = np.flatnonzero(stat.prev_used[l])
        nn = stat.N[l, S]
        has = S[nn > 1e-9]
        rn = float(stat.Rn[l])
        if len(has) == 0 and rn <= 1e-9:
            continue
        if len(has):
            acc = (stat.T[l, has] / stat.N[l, has][:, None]).sum(0)
            ks, km = (1.0 - s) / len(has), (s / rn if rn > 0 else 0.0)
        else:
            acc, ks, km = 0.0, 0.0, 1.0 / rn
        p[l] = np.minimum(1.0, acc * ks + stat.M[l] * km)
    return p


class Staging:
    """tier_spec_copy.hpp in python: per layer S staging slots; an entry lives A rounds; the oldest is evicted when full."""

    def __init__(self, nl, slots, age):
        self.slots, self.age = slots, age
        self.ent = [dict() for _ in range(nl)]       # layer -> {expert: [birth round, slot index]}
        self.copies = 0
        self.wasted = 0
        self.consumed = 0

    def tick(self, rnd):
        for d in self.ent:
            for e in [e for e, v in d.items() if rnd - v[0] > self.age]:
                del d[e]
                self.wasted += 1

    def stage(self, l, e, rnd):
        """Returns the staging slot to copy into, or None (already staged: kept alive)."""
        d = self.ent[l]
        if e in d:
            d[e][0] = rnd       # predicted again: keeps its slot, no new copy
            return None
        if len(d) >= self.slots:
            old = min(d, key=lambda x: d[x][0])
            slot = d[old][1]
            del d[old]
            self.wasted += 1
        else:
            slot = min(set(range(self.slots)) - {v[1] for v in d.values()})
        d[e] = [rnd, slot]
        self.copies += 1
        return slot

    def take(self, l, e):
        """Returns the staging slot of (l, e) and frees it, or None."""
        if e in self.ent[l]:
            slot = self.ent[l].pop(e)[1]
            self.consumed += 1
            return slot
        return None

    def leftover(self):
        return sum(len(d) for d in self.ent)


def parse(name):
    kind, _, arg = name.partition(":")
    return kind, (float(arg) if arg else 0.0)


def run(windows, init, pol, names, slots=4, max_copies=8, conf=1.0, age=4, nl=R.NL, ne=R.NE, seed=0, trans_w=1.0, sched=None):
    """Returns {name: per-window arrays} plus the baseline's per-window incoming count and swaps."""
    res = np.zeros((nl, ne), bool)
    res[init[:, 0], init[:, 1]] = True
    nw = len(windows)
    rng = np.random.default_rng(seed)
    stg = {nm: Staging(nl, slots, age) for nm in names}
    out = {nm: dict(inc=np.zeros(nw), hit=np.zeros(nw), copies=np.zeros(nw), made=np.zeros(nw)) for nm in names}
    base_swaps = np.zeros(nw)
    need_t = any(parse(nm)[0] in ("blt", "pst") for nm in names)
    stat = F.Stat(nl, ne, pol.decay, core_n=0, trans_w=trans_w) if need_t else None
    usage = np.zeros((nl, ne))
    rnd = 0
    hitmap = {}
    for wi, w in enumerate(windows):
        for l, ids in w:
            np.add.at(usage[l], ids.ravel(), 1.0)
        if (wi + 1) % pol.every != 0:
            continue
        rnd += 1
        raw = usage.copy()
        before = res.copy()
        n = R.adapt_round(res, usage, pol)
        ins = np.argwhere(res & ~before)                  # (layer, expert) incoming this round
        if sched is not None:                             # the schedule file: round, the tier's swaps (layer, in, out), later the staging
            outs = np.argwhere(before & ~res)
            sched.append(("R", rnd, sorted(map(tuple, ins)), sorted(map(tuple, outs))))
        base_swaps[wi] = len(ins)
        for nm in names:                                  # score the previous round's prediction against this round
            s = stg[nm]
            s.tick(rnd)
            out[nm]["inc"][wi] = len(ins)
            for l, e in ins:
                sl = s.take(int(l), int(e))
                if sl is not None:
                    out[nm]["hit"][wi] += 1
                if sched is not None and nm == names[0]:
                    hitmap.setdefault(rnd, {})[(int(l), int(e))] = sl
        usage *= pol.decay
        if stat is not None:
            slot = np.where(before, 0, -1)
            stat.observe(raw, slot, pol.decay)
            p = trans_p(stat)
        nxt = windows[wi + 1] if wi + 1 < nw else None
        for nm in names:
            kind, arg = parse(nm)
            up = usage.copy()
            if kind in ("blt", "pst") and stat is not None:
                up += p
            if kind in ("ps", "pst"):                # persistence: the next window routes like this one
                for l, ids in w:
                    np.add.at(up[l], ids.ravel(), 1.0)
            elif kind in ("la", "or") and nxt is not None:
                for l, ids in nxt:
                    x = ids.ravel()
                    if kind == "la":
                        x = x[rng.random(len(x)) < arg]
                    np.add.at(up[l], x, 1.0)
            r2 = res.copy()
            b2 = r2.copy()
            # every swap the tier would pick on the predicted counts (no cap: the cap is the staging budget below), by predicted score
            pol2 = R.Policy(every=pol.every, decay=pol.decay, swaps=10 ** 6, margin=pol.margin, mincand=pol.mincand,
                            mingain=0.0 if kind == "or" else conf)
            R.adapt_round(r2, up, pol2)
            pred = sorted(map(tuple, np.argwhere(r2 & ~b2)), key=lambda t: -up[t[0], t[1]])
            s = stg[nm]
            made = 0
            for l, e in pred:
                if made >= (10 ** 6 if kind == "or" else max_copies):
                    break
                sl = s.stage(int(l), int(e), rnd)
                if sl is not None:
                    made += 1
                    if sched is not None and nm == names[0]:
                        sched.append(("S", rnd, int(l), int(e), int(sl)))
            out[nm]["made"][wi] = made
            out[nm]["copies"][wi] = made
    if sched is not None:
        sched.append(("H", hitmap))
    return out, base_swaps, stg


def write_schedule(path, sched, slots, nl, ne, init):
    """Text schedule for the GPU test (tests of tier_spec_copy_gpu_test.cu).  Header: 'SCHED nl ne staging_slots_per_layer'.
    'R rnd' opens a round: 'I layer expert slot stage' = one incoming expert (slot = the victim's slot in its layer, stage = the
    staging slot it was staged in, or -1); 'S layer expert stageslot' = a staged copy issued at the END of the round.  Slot indexes
    follow the tier: each layer's residents get slots in profile order, a swap's incoming expert takes the slot of the victim it is
    paired with (inputs and outputs of a layer paired in expert order: any pairing is a valid placement)."""
    slot_of = {}
    cnt = np.zeros(nl, int)
    for l, e in init:
        slot_of[(int(l), int(e))] = int(cnt[l]); cnt[l] += 1
    hitmap = [x for x in sched if x[0] == "H"][0][1]
    lines = [f"SCHED {nl} {ne} {slots}"]
    cur = None
    for x in sched:
        if x[0] == "R":
            _, rnd, ins, outs = x
            lines.append(f"R {rnd}")
            by_l_in, by_l_out = {}, {}
            for l, e in ins: by_l_in.setdefault(l, []).append(e)
            for l, e in outs: by_l_out.setdefault(l, []).append(e)
            for l in sorted(by_l_in):
                for e_in, e_out in zip(by_l_in[l], by_l_out.get(l, [])):
                    sl = slot_of.pop((int(l), int(e_out)))
                    slot_of[(int(l), int(e_in))] = sl
                    st = hitmap.get(rnd, {}).get((int(l), int(e_in)))
                    lines.append(f"I {int(l)} {int(e_in)} {sl} {-1 if st is None else st}")
        elif x[0] == "S":
            _, rnd, l, e, sl = x
            lines.append(f"S {l} {e} {sl}")
    Path(path).write_text(chr(10).join(lines) + chr(10))


def ratio_ci(num, den, nboot=2000, block=8, seed=0):
    rng = np.random.default_rng(seed)
    n = len(num)
    sel = F.boot_idx(n, block, nboot, rng)
    x = num[sel].sum(1) / np.maximum(den[sel].sum(1), 1e-9)
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(num.sum() / max(den.sum(), 1e-9)), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("profile")
    ap.add_argument("slots_total", type=int)
    ap.add_argument("--variants", nargs="+", default=["bl", "blt", "ps", "pst", "la:0.6", "la:0.8", "or"])
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--swaps", type=int, default=32)
    ap.add_argument("--slots", type=int, default=4, help="staging slots per layer (STRATA_ADAPT_SPEC_SLOTS)")
    ap.add_argument("--max", type=int, default=8, dest="maxc", help="staged copies per round (STRATA_ADAPT_SPEC_MAX)")
    ap.add_argument("--conf", type=float, default=1.0, help="minimum predicted gain (STRATA_ADAPT_SPEC_CONF)")
    ap.add_argument("--age", type=int, default=4, help="rounds a staged expert is kept (STRATA_ADAPT_SPEC_AGE)")
    ap.add_argument("--bytes", type=float, default=None, help="bytes per expert (default: the IQ2_XS preset mean, derived)")
    ap.add_argument("--bw-h2d", type=float, default=13.4, help="host-to-device GB/s (T-193 link)")
    ap.add_argument("--bw-d2d", type=float, default=600.0, help="device copy GB/s (read + write)")
    ap.add_argument("--launch-us", type=float, default=10.0, help="per copy launch cost, us")
    ap.add_argument("--window-ms", type=float, default=24.8, help="decode window ms (T-016 short 24.81)")
    ap.add_argument("--sweep", action="store_true", help="also sweep slots 2/4/8, max 4/8/16/32 for blt and la:0.6")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--json", metavar="OUT")
    ap.add_argument("--schedule", metavar="OUT", help="write the first variant's round-by-round schedule (incoming, victims' slots, staging) for tier_spec_copy_gpu_test")
    ap.add_argument("--max-rounds", type=int, default=0, help="with --schedule: keep only the first N rounds")
    a = ap.parse_args()
    windows, k, starts, weights, total = F.load_dump(a.trace)
    init = R.load_profile(a.profile, a.slots_total)
    pol = R.Policy(every=a.every, swaps=a.swaps)
    by = a.bytes if a.bytes else mean_bytes()
    t_h = by / (a.bw_h2d * 1e9) * 1e3 + a.launch_us / 1e3          # ms
    t_d = by / (a.bw_d2d * 1e9) * 1e3 + a.launch_us / 1e3
    print(f"{len(windows)} windows, k={k}, cache slots {a.slots_total}, round every {a.every} window(s), {a.swaps} swaps cap; staging: {a.slots} slots/layer, "
          f"{a.maxc} copies/round, conf {a.conf}, age {a.age} rounds; expert {by:,.0f} B -> H2D {t_h*1e3:.0f} us ({a.bw_h2d} GB/s + {a.launch_us:.0f} us), "
          f"D2D {t_d*1e3:.0f} us; window {a.window_ms} ms")
    runs = [(a.slots, a.maxc)]
    names = list(a.variants)
    results = {}
    if a.sweep:
        runs = [(s, m) for s in (2, 4, 8) for m in (4, 8, 16, 32)]
    table = {}
    for (S, M) in runs:
        nms = names if not a.sweep else ["blt", "ps", "pst", "la:0.6"]
        sched = [] if (a.schedule and not a.sweep) else None
        out, bsw, stg = run(windows, init, pol, nms, slots=S, max_copies=M, conf=a.conf, age=a.age, sched=sched)
        if sched is not None:
            write_schedule(a.schedule, sched, S, R.NL, R.NE, init)
        n_in = float(bsw.sum())
        print(f"\nstaging {S}/layer, {M} copies/round: baseline swaps {n_in / len(windows):.2f} per window ({int(n_in)} in all)")
        print(f"{'variant':9} {'hit rate':>8} {'[95% CI]':>15} {'staged/win':>10} {'waste':>7} {'[95% CI]':>15} {'PCIe bytes x':>12} {'D2D-finalised/win':>17} {'saving ms/win':>13} {'[95% CI]':>15} {'% of window':>11}")
        for nm in nms:
            o = out[nm]
            hr, hlo, hhi = ratio_ci(o["hit"], o["inc"], a.nboot)
            made = o["made"]
            hits = o["hit"]
            # waste = staged copies that no real incoming expert used (aged out, evicted, or still staged at the end), over staged copies
            s = stg[nm]
            waste = 1.0 - hits.sum() / max(made.sum(), 1)
            sel = F.boot_idx(len(made), 8, a.nboot, np.random.default_rng(1))
            wb = 1.0 - hits[sel].sum(1) / np.maximum(made[sel].sum(1), 1e-9)
            wlo, whi = (float(x) for x in np.percentile(wb, [2.5, 97.5]))
            pc = (made.sum() + (o["inc"].sum() - hits.sum())) / max(o["inc"].sum(), 1)
            sav = hits * (t_h - t_d)
            sv, slo, shi = ratio_ci(sav, np.ones(len(sav)), a.nboot)
            print(f"{nm:9} {hr:8.3f} [{hlo:5.3f},{hhi:5.3f}] {made.sum()/len(made):10.2f} {waste:7.3f} [{wlo:5.3f},{whi:5.3f}] {pc:12.2f} {hits.sum()/len(hits):17.2f} {sv:13.3f} [{slo:5.3f},{shi:5.3f}] {100*sv/a.window_ms:10.2f}%")
            table[f"{nm} S{S} M{M}"] = dict(hit=hr, hit_lo=hlo, hit_hi=hhi, staged_per_win=float(made.sum() / len(made)), waste=waste,
                                          pcie_x=float(pc), d2d_per_win=float(hits.sum() / len(hits)), saving_ms=sv, saving_lo=slo, saving_hi=shi,
                                          saving_pct=100 * sv / a.window_ms, leftover=s.leftover())
    if a.json:
        Path(a.json).write_text(json.dumps(dict(windows=len(windows), k=k, bytes=by, t_h2d_ms=t_h, t_d2d_ms=t_d, params=vars(a), table=table), indent=1))


if __name__ == "__main__":
    main()
