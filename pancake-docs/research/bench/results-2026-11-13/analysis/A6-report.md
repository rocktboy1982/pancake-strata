# A6: time-model refit check (CPU only, 2026-11-13)

Saved by the main session from the agent's final text (the agent's own write was refused). Other files are under /home/dan/agent-results/A6/.

## Verdict
- Reproduced: yes. The committed inputs reproduce the 2026-11-10 A6 numbers exactly. The "8.1 points" is the rank-guard arm (TOP=2): model +12.04% vs measured +20.1 ± 3.3, error −8.06.
- Held-out error on the three held-out arms (RG, KMISS 2,8, FUSE_PHASES): mean 3.38 points, worst 8.06.
- Against the 5–9 target: the mean is inside it. The worst arm (8.1) is at the upper edge, not below 5.
- Refit: no honest improvement. Refitting the deferred-layer cost (sdef) alone reproduces the published fit. Refitting all five parameters to all five arms fits every arm but is in-sample. Leave-one-arm-out errors are −2.8, +4.2, +2.9, −0.8 (mean about 2.7), and the fitted parameters swing widely, so the data do not identify them.

## What the model is
`tools/pancake/time_ceiling.py` (`--validate`): a 48-layer chain per decode window: GPU doorbell (mixer and router), CPU pool time (missed experts × c, pool barriers, promotion DMA contention), GPU combine. Under deferral a layer waits only for the heavy share of its CPU rows and pays a fitted per-layer cost sdef. Edit window only; decode only.

Published parameters: c 44.6 µs, pre-doorbell GPU 15.75 ms, mu 0.65, E 4.3 ms, ps 40 µs, alpha 0.5, heavy 0.4 (0.55 under TOP=2), sdef 91.2 µs, extra +1.05 ms.

## Inputs
- Trace: `/home/dan/clone-of-d/src/Strata/bench/prefetch/routing.bin` (219 windows). Its weight column is all zeros, so the heavy share (0.4) is a prior, not measured.
- Profile: `/home/dan/clone-of-d/src/Strata/data/expert-profile-pancake.bin`.
- Measured arms (`bench/results/2026-11-09-test-phase/RESULTS.md`, 95% intervals): D1 deferral +15.8 ± 3.9 (fit point); D2 +17.3 ± 2.9; RG rank guard +20.1 ± 3.3; K28 +2.0 ± 1.6; FU FUSE_PHASES +2.5 ± 1.7.

## Numbers
- Published, reproduced: D1 +15.80 (error 0.00); D2 +15.80 vs +17.3 (−1.50); RG +12.04 vs +20.1 (−8.06); K28 +2.69 vs +2.0 (+0.69); FU +1.13 vs +2.5 (−1.37).
- Refit of sdef on D1 alone: 91.2 µs, no change.
- In-sample, all five parameters on all five arms: alpha 0.462, heavy 0.440, top2 0.181, sdef 88.7 µs, ps 81.5 µs. Fits every arm, but this is not a validation.
- Leave-one-out: D1 held out → +12.98 (error −2.82); RG → +24.26 (+4.16); K28 → +4.93 (+2.93); FU → +1.70 (−0.80).
- RG sensitivity to top2: at 0.00–0.20 the model gives +17.76; at 0.40, +15.80; at 0.55, +12.04. Even at top2 = 0 the model is 2.3 points short of +20.1.
- Rank-guard gap: measured RG − D1 = +4.3 ± 5.1; model = −3.8. The gap is about 3.1 standard errors of the difference. The plan's "1.1 sigma, may be noise" tests the measured gap against zero, not the model, so the miss is probably not noise.

## Night queue (not in the fit)
Deferral on vs off: iq2xs overall +1.0%, code +7.8%, prose +3.2%; q2 overall −8.1%, code +2.3%, prose −7.7%. Five prompts, not an edit window, different models. The sign depends on model and workload, which the single fitted edit point does not capture.

## Missing data to close the gap
1. A routing trace with non-zero weights (the heavy share is a prior).
2. A per-layer late-wait profile to pin the fitted 91 µs deferred-layer cost.
3. A 5-repeat rerun of deferral vs TOP=2 on edits.
4. A short-chat routing trace.
5. A TTFT term, or a TTFT measurement. The model has none.
