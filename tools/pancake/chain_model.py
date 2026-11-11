"""chain_model.py - the A6 engine-chain time model (time_ceiling.window_ms2) with its fitted constants, as one function.

time_ceiling.py --validate fits the model to the edit window of the 7900 XT fast-hip preset (bench/results/2026-11-10-offline/
A6_A17.md) and prices policies on it; its constants live inside validate_report.  Tools that price a policy on other traces
(selftune.py, the bootstrap check) need the same constants without running the whole report, so they are kept here, once.

    run = tc.tier(windows, init, rp.Policy(every=1, swaps=32, decay=0.7))     # (hit share, misses, promotions, per-layer misses)
    ms = chain_model.window_ms(run)                                           # mean modeled decode-window ms
    gain = chain_model.gain_pct(chain_model.window_ms(base_run), ms)

What the constants are (all from `time_ceiling.py --validate` defaults, trace = 219-window edit window, 81.2% hits):
  pre 15.75 ms   GPU work before the router doorbells, mu 0.65 of it the layer's mixer (profile-fasthip.txt)
  E 4.3 ms       GPU work on the VRAM experts, hidden behind the CPU
  ps / sync      40 us pool barriers inside the CPU time / 40 us combine sync per layer
  c 44.6 us      CPU pool time per distinct missed expert (48.7 us per miss incl. barriers, equal in the edit and short windows)
  alpha 0.5      CPU ms lost per ms of promotion DMA (DDR4 shared with the pool: about the 12 / 27 GB/s ratio)
  SDEF 91 us     per-layer cost of a deferred layer, the one fitted number (to the measured deferral +15.8%)
  HEAVY 0.4      share of a layer's CPU time kept on time under STRATA_VERIFY_DEFER_CPU (BELOW 0.1; a prior)
  EXTRA 1.05 ms  the bench's window is longer than the profile's (host stalls at window edges)
Validity: the model is an EDIT-window model.  Out of sample it predicted KMISS 2,8 +2.7% (measured +2.0 +-1.6), FUSE_PHASES
+1.1% (+2.5 +-1.7) and missed the rank guard (+12.0% against +20.1 +-3.3); short chat has no routing trace to validate on.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import time_ceiling as tc  # noqa: E402

PRESET = dict(pre=15.75, mu=0.65, E=4.3, ps=0.040, sync=0.040, draft=2.5, c=0.0446, alpha=0.5, dma_per=1.4 / 12.0)
HEAVY, SDEF, EXTRA = 0.4, 0.091, 1.05


def window_ms(run, mode="defer", fuse_phases=True, c_us=None):
    """Mean modeled window ms of one time_ceiling run tuple (hit, miss/window, promotions/window, per-layer misses).

    mode "defer" is the fast-hip preset (STRATA_VERIFY_DEFER_CPU=1), "serial" a preset without deferral; fuse_phases is
    STRATA_POOL_FUSE_PHASES=1 (one pool phase barrier instead of two).  c_us overrides the CPU time per missed expert."""
    p = PRESET if c_us is None else {**PRESET, "c": c_us / 1e3}
    cut = p["ps"] / 2 if fuse_phases else 0.0
    return float(np.mean([tc.window_ms2(ml, pr, p, mode, HEAVY, SDEF, cut) for ml, pr in zip(run[3], run[2])])) + EXTRA


def gain_pct(base_ms, ms):
    """Decode-speed gain of a window time against the base's, in percent (the abtest.py convention: tok/s ratio - 1)."""
    return (base_ms / ms - 1.0) * 100.0
