"""selftune.py - pancake-strata closed-loop self-tuning from the owner's own sessions (MOD_PLAN item 70, phase 14).

One pass of the loop, offline (no GPU):
  1. traces: window traces recorded during normal use (STRATA_WINDOW_TRACE=<file>, item 64) and/or --dump-routing files;
     one file = one session.  Only decode windows count (at most MAX_ROWS rows; a merged prompt window has thousands).
  2. a candidate expert profile: how often each (layer, expert) was routed, blended with the current ranking (--blend: the
     weight of the current ranking, so one odd session cannot flip it), every pair ranked (a profile that ranks fewer
     pairs than a card can hold caps the cache, issue #46).
  3. held-out check: leave one session out (several files), or k contiguous blocks of the one session (--folds).  For each
     fold the candidate is learned on the other folds and priced on the held-out one, against the current profile, on
     the A6 engine-chain time model (chain_model.py: promotion DMA costs CPU time, deferral gates the next router) with the
     base config's own mode (deferral and FUSE_PHASES read from its env).  Priced: the preset tier and k-th-miss 2/8 on
     the current and on the candidate profile.  A policy goes on only if its mean gain over the current profile on the
     preset tier is >= --min-gain % AND at least --min-positive of the folds gain: one session is not a verdict.
  4. it writes the candidate profile (learned on everything) and an A/B spec for tools/pancake/abtest.py; the test phase
     measures it on the real engine (then quality_gate.py: expert placement moves work between GPU and CPU kernels, whose
     rounding differs, so the spec is marked lossy) before anything changes in a preset.

    python tools/pancake/selftune.py --traces a.stwt b.stwt [--routing old.bin] --profile data/expert-profile-pancake.bin \
        --slots 10909 --out bench/selftune/<day> [--folds 4] [--blend 0.3] [--min-gain 2] [--min-positive 0.75]

Limits (measured on the one recorded trace, bench/prefetch/routing.bin, 219 windows, 4 contiguous blocks; see
pancake-docs/batch3/selftune.md): the gain of a candidate varies from -3% to +12% between blocks, grows with the learning
windows (27 windows: +1.7%, 54: +4.7%, 109: +9.7%, 191: +9.9%) and is positive in only ~3 of 4 held-out blocks, so the gate wants
>= --min-windows windows.  The model is an edit-window model (short chat has no validated trace).  The profile matters most at
a cold start: the adaptive tier moves 32 experts per window and forgets the profile within the session.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "trace"))
sys.path.insert(0, str(ROOT / "tools"))
import chain_model as cm  # noqa: E402
import make_profile as mp  # noqa: E402
import replay_cache_policy as rp  # noqa: E402
import time_ceiling as tc  # noqa: E402
import wtrace  # noqa: E402

MAX_ROWS = 16   # a decode window is at most --lookup-max-t 12 rows (spec 4 + drafts); a merged prompt window has thousands
BASELINE = "current tier"
POLICIES = ("candidate tier", "current kmiss 2/8", "candidate kmiss 2/8")


def read_session(path, dump):
    """The decode windows of one trace file as the [(layer, ids)] lists the replay tools take."""
    tr = wtrace.from_dump_routing(path) if dump else wtrace.read(path)
    return [w for w in wtrace.to_replay_windows(tr) if len(w) == rp.NL and w[0][1].shape[0] <= MAX_ROWS]


def load_sessions(traces, routings):
    return [(Path(p).name, read_session(p, False)) for p in traces] + [(Path(p).name, read_session(p, True)) for p in routings]


def make_folds(sessions, k=4, min_fold=20):
    """(name, learn, judge) per fold: leave one session out, or k contiguous blocks of the single session.
    A session shorter than min_fold windows is never judged on, only learned from."""
    if len(sessions) == 1:
        name, w = sessions[0]
        blocks = [w[i * len(w) // k:(i + 1) * len(w) // k] for i in range(k)]
        names = [f"{name}#{i + 1}/{k}" for i in range(k)]
    else:
        names, blocks = [n for n, _ in sessions], [w for _, w in sessions]
    return [(names[i], [w for j, b in enumerate(blocks) if j != i for w in b], judge)
            for i, judge in enumerate(blocks) if len(judge) >= min_fold]


def candidate_ranking(windows, current, blend):
    """Every (layer, expert) pair, best first: (1 - blend) x routing share + blend x the current ranking (linear in rank)."""
    freq = defaultdict(float)
    for w in windows:
        for l, ids in w:
            for e in ids.ravel().tolist():
                freq[(l, int(e))] += 1.0
    total = sum(freq.values()) or 1.0
    n = len(current)
    score = defaultdict(float)
    for (l, e), c in freq.items():
        score[(l, e)] += (1 - blend) * c / total
    for i, (l, e) in enumerate(current):   # the current ranking as a prior: linear weight by rank
        score[(int(l), int(e))] += blend * 2.0 * (n - i) / (n * (n + 1))
    ranked = sorted(score, key=lambda x: (-score[x], x))
    return mp.rank_profile(ranked, {})[0]   # + every pair the trace and the current ranking both missed


def policy_runs(windows, init):
    """time_ceiling run tuples of the preset tier and k-th-miss 2/8 for one starting profile."""
    budgets = np.bincount(init[:, 0], minlength=rp.NL)
    inits = [[int(e) for ll, e in init if ll == l] for l in range(rp.NL)]
    seqs, uses = tc.per_layer(windows)
    return {"tier": tc.tier(windows, init, rp.Policy(every=1, swaps=32, decay=0.7)),
            "kmiss 2/8": tc.replay(seqs, uses, budgets, inits, "kmiss", kmiss=(2, 8))}


def fold_gains(learn, judge, current, cur_init, slots, blend, mode, fuse, c_us):
    """Modeled gain (%) over the current profile on the preset tier, for each policy, on one held-out fold."""
    cand_init = np.array(candidate_ranking(learn, current, blend)[:slots])
    cur, cand = policy_runs(judge, cur_init), policy_runs(judge, cand_init)

    def ms(run):
        return cm.window_ms(run, mode, fuse, c_us)

    base = ms(cur["tier"])
    runs = {BASELINE: cur["tier"], "candidate tier": cand["tier"], "current kmiss 2/8": cur["kmiss 2/8"],
            "candidate kmiss 2/8": cand["kmiss 2/8"]}
    return {name: cm.gain_pct(base, ms(r)) for name, r in runs.items()}, {name: float(r[0]) for name, r in runs.items()}


def verdict(per_fold, min_gain, min_positive):
    """Per policy over the folds: mean, worst, share of folds that gain; ok = mean >= min_gain and enough folds gain."""
    out = {}
    for name, g in per_fold.items():
        g = np.array(g, float)
        pos = float((g > 0).mean())
        out[name] = {"mean": round(float(g.mean()), 2), "min": round(float(g.min()), 2), "max": round(float(g.max()), 2),
                     "positive": round(pos, 2), "folds": [round(float(x), 2) for x in g],
                     "ok": bool(g.mean() >= min_gain and pos >= min_positive)}
    return out


def make_spec(best, cand_path, base_config):
    """The abtest.py spec: the current preset against the winning policy (None: nothing passed, only the base)."""
    arms = {"current": {}}
    if best is not None:
        arm = {}
        if best.startswith("candidate"):
            arm["args"] = {"--expert-profile": str(Path(cand_path).resolve())}
        if "kmiss" in best:
            arm["env"] = {"STRATA_ADAPT_KMISS": "2,8"}   # the blocking tier: --adapt-async was measured and rejected
        arms["selftune"] = arm
    return {"base": base_config, "arms": arms, "cases": ["short", "edit"], "reps": 3, "lossy": True,
            "note": "tools/pancake/selftune.py (MOD_PLAN item 70). Placement change: no prompt-reading case (long* is unaffected); "
                    "lossy because experts move between GPU and CPU kernels whose rounding differs (a different text is "
                    "placement-dependent, not a bug): run quality_gate.py after the screen."}


def preset_mode(base_config):
    """(mode, fuse_phases, file) the time model should use for a base config: deferral and FUSE_PHASES from its env.
    Without the file: the current fast-hip preset."""
    base = Path(base_config)
    for p in (ROOT / base, ROOT / "configs/pancake" / base.name.replace("pancake-", "", 1)):
        if p.exists():
            env = json.loads(p.read_text()).get("env", {})
            return ("defer" if env.get("STRATA_VERIFY_DEFER_CPU") == "1" else "serial",
                    env.get("STRATA_POOL_FUSE_PHASES") == "1", str(p))
    return "defer", True, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", nargs="*", default=[], help="window traces (.stwt), one session each")
    ap.add_argument("--routing", nargs="*", default=[], help="--dump-routing files, one session each")
    ap.add_argument("--profile", required=True, help="the current expert profile (the base config's --expert-profile)")
    ap.add_argument("--slots", type=int, required=True, help="VRAM expert slots (the engine's 'expert cache' line)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=4, help="blocks when there is only one session")
    ap.add_argument("--blend", type=float, default=0.3)
    ap.add_argument("--min-gain", type=float, default=2.0, help="mean modeled gain (%%) a policy needs (the screen's promote line)")
    ap.add_argument("--min-positive", type=float, default=0.75, help="share of held-out folds that must gain")
    ap.add_argument("--min-windows", type=int, default=200, help="decode windows below which nothing is judged")
    ap.add_argument("--min-fold", type=int, default=20, help="windows a held-out fold needs")
    ap.add_argument("--c-us", type=float, default=None, help="CPU us per missed expert (default: the A6 fit, 44.6)")
    ap.add_argument("--base-config", default="pancake-flashnext-iq2_xs-fast-hip.json")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    sessions = load_sessions(a.traces, a.routing)
    total = sum(len(w) for _, w in sessions)
    if total < a.min_windows:
        print(f"only {total} decode windows in the traces (need {a.min_windows}): record more before tuning")
        return 2
    folds = make_folds(sessions, a.folds, a.min_fold)
    if len(folds) < 2:
        print(f"only {len(folds)} held-out folds with >= {a.min_fold} windows: need two")
        return 2
    mode, fuse, found = preset_mode(a.base_config)
    print(f"{total} decode windows, {len(sessions)} session(s), {len(folds)} held-out folds; model: {mode} deferral, "
          f"FUSE_PHASES {'on' if fuse else 'off'} ({found or 'base config not found: the fast-hip preset assumed'})")
    if len(sessions) == 1:
        print("  one session: its blocks are not independent sessions; record more sessions before trusting a gain")

    current = mp.read_profile(a.profile)
    cur_init = rp.load_profile(a.profile, a.slots)
    per_fold, rows = {p: [] for p in POLICIES}, []
    for name, learn, judge in folds:
        gains, hits = fold_gains(learn, judge, current, cur_init, a.slots, a.blend, mode, fuse, a.c_us)
        for p in POLICIES:
            per_fold[p].append(gains[p])
        rows.append({"fold": name, "learn": len(learn), "judge": len(judge), "hit_current": round(hits[BASELINE], 4),
                     "gain_pct": {p: round(gains[p], 2) for p in POLICIES}})
        print(f"  {name:28s} learn {len(learn):4d} judge {len(judge):4d} hit {hits[BASELINE]:.1%}  "
              + "  ".join(f"{p.replace('candidate', 'cand').replace('current', 'cur')} {gains[p]:+6.1f}%" for p in POLICIES))
    v = verdict(per_fold, a.min_gain, a.min_positive)
    for p in POLICIES:
        print(f"{p:22s} mean {v[p]['mean']:+6.2f}%  worst {v[p]['min']:+6.2f}%  folds gaining {v[p]['positive']:.0%}  "
              f"{'GOES TO THE A/B' if v[p]['ok'] else 'no'}")
    passed = [p for p in POLICIES if v[p]["ok"]]
    best = max(passed, key=lambda p: v[p]["mean"]) if passed else None

    cand_path = out / "candidate-profile.bin"
    everything = [win for _, wins in sessions for win in wins]
    mp.write_profile(cand_path, candidate_ranking(everything, current, a.blend))
    spec = make_spec(best, cand_path, a.base_config)
    (out / "abtest-spec.json").write_text(json.dumps(spec, indent=1))
    report = {"windows": total, "sessions": [{"name": n, "windows": len(w)} for n, w in sessions], "model": {
        "mode": mode, "fuse_phases": fuse, "base_config": found, "c_us": a.c_us}, "folds": rows, "verdict": v, "best": best,
        "min_gain": a.min_gain, "min_positive": a.min_positive}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"{'candidate goes to the A/B: ' + best if best else 'nothing above the gate: nothing to A/B'}\n"
          f"written: {cand_path}, {out / 'abtest-spec.json'}, {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
